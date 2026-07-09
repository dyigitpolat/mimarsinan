"""Graphviz writer."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.common.presentation import safe_float

from mimarsinan.visualization.graphviz.common import _percent, _compress_ranges, _truncate, _dot_html_label

def write_softcore_mapping_dot(
    soft_core_mapping: Any,
    out_dot: str,
    *,
    title: str | None = None,
    cluster_by_psum_group: bool = True,
    max_edges_per_node: int = 64,
) -> None:
    """Visualize a mapped SoftCoreMapping."""
    os.makedirs(os.path.dirname(out_dot) or ".", exist_ok=True)

    cores = list(getattr(soft_core_mapping, "cores", []))
    out_sources = list(getattr(soft_core_mapping, "output_sources", []))

    lines: list[str] = []
    lines.append("digraph SoftCoreMapping {")
    lines.append("  rankdir=LR;")
    lines.append("  graph [fontname=\"Helvetica\", fontsize=10, labelloc=t];")
    lines.append("  node [shape=plaintext, fontname=\"Helvetica\"];")
    lines.append("  edge [fontname=\"Helvetica\", fontsize=9];")

    if title:
        lines.append(f"  label=\"{_truncate(title, max_chars=140)}\";")

    lines.append("  input [label=< <B>INPUT</B> >, shape=plaintext];")
    lines.append("  const1 [label=< <B>CONST(1)</B> >, shape=plaintext];")
    lines.append("  output [label=< <B>OUTPUT</B> >, shape=plaintext];")

    by_group: dict[int | None, list[Any]] = {}
    for c in cores:
        by_group.setdefault(getattr(c, "psum_group_id", None), []).append(c)

    def _emit_core_node(c: Any):
        cid = int(getattr(c, "id"))
        nid = f"c{cid}"

        mat = getattr(c, "core_matrix")
        ax = int(mat.shape[0])
        neu = int(mat.shape[1])
        nnz = int(np.count_nonzero(mat))
        total = int(ax * neu)

        name = getattr(c, "name", None) or f"core_{cid}"
        psum = ""
        if getattr(c, "psum_group_id", None) is not None or getattr(c, "psum_role", None) is not None:
            psum = f"group={getattr(c,'psum_group_id',None)}, role={getattr(c,'psum_role',None)}"

        rows = [
            ("id", str(cid)),
            ("type", "SoftCore"),
            ("shape(axons×neurons)", f"{ax}×{neu}"),
            ("weights nnz", f"{nnz}/{total} ({_percent(nnz, total)})"),
            ("threshold", str(safe_float(getattr(c, "threshold", None)) or "n/a")),
            ("latency", str(getattr(c, "latency", None) if getattr(c, "latency", None) is not None else "n/a")),
        ]
        if psum:
            rows.append(("psum", psum))

        role = getattr(c, "psum_role", None)
        if role == "partial_pos":
            color = "#D5F5E3"
        elif role == "partial_neg":
            color = "#FADBD8"
        elif role == "accum":
            color = "#FCF3CF"
        else:
            color = "#EBF5FB"

        label = _dot_html_label(rows, title=str(name), color=color)
        lines.append(f"  {nid} [label={label}];")

    if cluster_by_psum_group and any(g is not None for g in by_group):
        for gid, group_cores in by_group.items():
            if gid is None:
                for c in group_cores:
                    _emit_core_node(c)
                continue
            lines.append(f"  subgraph cluster_psum_{int(gid)} {{")
            lines.append("    style=rounded; color=\"#AAB7B8\";")
            lines.append(f"    label=\"psum_group {int(gid)}\";")
            for c in group_cores:
                _emit_core_node(c)
            lines.append("  }")
    else:
        for c in cores:
            _emit_core_node(c)

    for c in cores:
        cid = int(getattr(c, "id"))
        tgt = f"c{cid}"
        axon_sources: list[SpikeSource] = list(getattr(c, "axon_sources", []))

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
                src_node = f"c{src_core}"

            label = (
                f"axons:{_compress_ranges(entry['tgt_ax'])}\\n"
                f"idx:{_compress_ranges(entry['src_idx'])}\\n"
                f"n={len(entry['tgt_ax'])}"
            )
            lines.append(f"  {src_node} -> {tgt} [label=\"{_truncate(label, max_chars=220)}\"];")

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
            src_node = f"c{src_core}"

        label = f"out:{_compress_ranges(entry['out_i'])}\\nidx:{_compress_ranges(entry['src_idx'])}"
        lines.append(f"  {src_node} -> output [label=\"{_truncate(label, max_chars=220)}\"];")

    lines.append("}")

    with open(out_dot, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
