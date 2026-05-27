"""Graphviz writer."""

from __future__ import annotations

import html
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRNode, IRSource, NeuralCore
from mimarsinan.mapping.softcore_mapping import HardCoreMapping
from mimarsinan.common.layer_key import layer_key_from_node_name
from mimarsinan.common.safe_numeric import safe_float

import re


from mimarsinan.visualization.graphviz.common import try_render_dot, _embed_svg_images, _percent, _compress_ranges, _truncate, _dot_html_label, _dot_html_label_mixed, _stack_sample_lines

def write_hardcore_mapping_dot(
    hard_core_mapping: HardCoreMapping,
    out_dot: str,
    *,
    title: str | None = None,
    max_edges_per_node: int = 128,
) -> None:
    os.makedirs(os.path.dirname(out_dot) or ".", exist_ok=True)

    cores = list(hard_core_mapping.cores)
    out_sources = list(hard_core_mapping.output_sources.flatten())

    # Invert neuron_mapping for "packed softcore ids" summary
    packed: dict[int, dict[int, list[int]]] = {}
    # packed[hard_core_idx][soft_core_id] = [hard_neuron_idxs...]
    for (soft_core_id, soft_neuron), (hard_core_idx, hard_neuron) in hard_core_mapping.neuron_mapping.items():
        packed.setdefault(int(hard_core_idx), {}).setdefault(int(soft_core_id), []).append(int(hard_neuron))

    lines: list[str] = []
    lines.append("digraph HardCoreMapping {")
    lines.append("  rankdir=LR;")
    lines.append("  graph [fontname=\"Helvetica\", fontsize=10, labelloc=t];")
    lines.append("  node [shape=plaintext, fontname=\"Helvetica\"];")
    lines.append("  edge [fontname=\"Helvetica\", fontsize=9];")

    if title:
        lines.append(f"  label=\"{_truncate(title, max_chars=140)}\";")

    lines.append("  input [label=< <B>INPUT</B> >, shape=plaintext];")
    lines.append("  const1 [label=< <B>CONST(1)</B> >, shape=plaintext];")
    lines.append("  output [label=< <B>OUTPUT</B> >, shape=plaintext];")

    for i, core in enumerate(cores):
        nid = f"h{i}"
        ax = int(core.axons_per_core)
        neu = int(core.neurons_per_core)
        used_ax = int(ax - core.available_axons)
        used_neu = int(neu - core.available_neurons)
        nnz = int(np.count_nonzero(core.core_matrix))
        total = int(ax * neu)

        rows = [
            ("id", str(i)),
            ("type", "HardCore"),
            ("capacity(axons×neurons)", f"{ax}×{neu}"),
            ("used(axons×neurons)", f"{used_ax}×{used_neu}"),
            ("weights nnz", f"{nnz}/{total} ({_percent(nnz, total)})"),
            ("threshold", str(safe_float(core.threshold) if core.threshold is not None else "n/a")),
            ("latency", str(core.latency if core.latency is not None else "n/a")),
            ("unusable_space", str(int(getattr(core, "unusable_space", 0)))),
        ]

        # Packed softcores summary (by soft core id -> neuron ranges)
        soft_ids = packed.get(i, {})
        if soft_ids:
            # Keep it readable.
            items = []
            for sid, hard_neus in sorted(soft_ids.items())[:10]:
                items.append(f"{sid}:{_compress_ranges(hard_neus, max_ranges=6)}")
            if len(soft_ids) > 10:
                items.append(f"...(+{len(soft_ids)-10})")
            rows.append(("packed_softcores", _truncate(" ".join(items), max_chars=220)))

        label = _dot_html_label(rows, title=f"HardCore {i}", color="#E8DAEF")
        lines.append(f"  {nid} [label={label}];")

    # Edges from axon_sources (compressed)
    for i, core in enumerate(cores):
        tgt = f"h{i}"
        axon_sources: list[SpikeSource] = list(core.axon_sources)

        by_src: dict[tuple[str, int], dict[str, list[int]]] = {}
        for ax_idx, src in enumerate(axon_sources):
            if src.is_off_:
                continue
            if src.is_input_:
                key = ("input", -2)
            elif src.is_always_on_:
                key = ("const1", -3)
            else:
                key = ("core", int(src.core_))

            entry = by_src.setdefault(key, {"tgt_ax": [], "src_idx": []})
            entry["tgt_ax"].append(int(ax_idx))
            entry["src_idx"].append(int(src.neuron_))

        keys = list(by_src.keys())
        if len(keys) > max_edges_per_node:
            keys = keys[:max_edges_per_node]

        for key in keys:
            kind, src_core = key
            entry = by_src[key]
            if kind == "input":
                src_node = "input"
            elif kind == "const1":
                src_node = "const1"
            else:
                src_node = f"h{src_core}"

            label = (
                f"axons:{_compress_ranges(entry['tgt_ax'])}\\n"
                f"idx:{_compress_ranges(entry['src_idx'])}\\n"
                f"n={len(entry['tgt_ax'])}"
            )
            lines.append(f"  {src_node} -> {tgt} [label=\"{_truncate(label, max_chars=220)}\"];")

    # Outputs
    by_src_out: dict[tuple[str, int], dict[str, list[int]]] = {}
    for out_i, src in enumerate(out_sources):
        if src.is_off_:
            continue
        if src.is_input_:
            key = ("input", -2)
        elif src.is_always_on_:
            key = ("const1", -3)
        else:
            key = ("core", int(src.core_))

        entry = by_src_out.setdefault(key, {"out_i": [], "src_idx": []})
        entry["out_i"].append(int(out_i))
        entry["src_idx"].append(int(src.neuron_))

    for key, entry in by_src_out.items():
        kind, src_core = key
        if kind == "input":
            src_node = "input"
        elif kind == "const1":
            src_node = "const1"
        else:
            src_node = f"h{src_core}"

        label = f"out:{_compress_ranges(entry['out_i'])}\\nidx:{_compress_ranges(entry['src_idx'])}"
        lines.append(f"  {src_node} -> output [label=\"{_truncate(label, max_chars=220)}\"];")

    lines.append("}")

    with open(out_dot, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

