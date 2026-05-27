"""IR graph DOT writers."""

from __future__ import annotations

import html
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRNode, IRSource, NeuralCore
from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.common.layer_key import layer_key_from_node_name
from mimarsinan.common.safe_numeric import safe_float

import re


from mimarsinan.visualization.graphviz.common import (
    _compress_ranges,
    _dot_html_label,
    _dot_html_label_mixed,
    _percent,
    _stack_sample_lines,
    _truncate,
)

def write_ir_graph_dot(
    ir_graph: IRGraph,
    out_dot: str,
    *,
    title: str | None = None,
    max_edges_per_node: int = 64,
) -> None:
    os.makedirs(os.path.dirname(out_dot) or ".", exist_ok=True)

    lines: list[str] = []
    lines.append("digraph IRGraph {")
    lines.append("  rankdir=LR;")
    lines.append("  graph [fontname=\"Helvetica\", fontsize=10, labelloc=t];")
    lines.append("  node [shape=plaintext, fontname=\"Helvetica\"];")
    lines.append("  edge [fontname=\"Helvetica\", fontsize=9];")

    if title:
        lines.append(f"  label=\"{_truncate(title, max_chars=140)}\";")

    # Special nodes
    lines.append("  input [label=< <B>INPUT</B> >, shape=plaintext];")
    lines.append("  const1 [label=< <B>CONST(1)</B> >, shape=plaintext];")
    lines.append("  output [label=< <B>OUTPUT</B> >, shape=plaintext];")

    # Nodes
    for node in ir_graph.nodes:
        nid = f"n{int(node.id)}"
        if isinstance(node, NeuralCore):
            _mat = node.get_core_matrix(ir_graph)
            ax = int(_mat.shape[0])
            neu = int(_mat.shape[1])
            nnz = int(np.count_nonzero(_mat))
            total = int(ax * neu)
            rows = [
                ("id", str(node.id)),
                ("type", "NeuralCore"),
                ("shape(axons×neurons)", f"{ax}×{neu}"),
                ("weights nnz", f"{nnz}/{total} ({_percent(nnz, total)})"),
                ("threshold", str(safe_float(node.threshold) if node.threshold is not None else "n/a")),
                ("latency", str(node.latency if node.latency is not None else "n/a")),
            ]
            if node.psum_group_id is not None or node.psum_role is not None:
                rows.append(("psum", f"group={node.psum_group_id}, role={node.psum_role}"))
            label = _dot_html_label(rows, title=str(node.name), color="#D6EAF8")
        elif isinstance(node, ComputeOp):
            rows = [
                ("id", str(node.id)),
                ("type", "ComputeOp"),
                ("op_type", str(node.op_type)),
                ("input_shape", str(node.input_shape)),
                ("output_shape", str(node.output_shape)),
                ("params", _truncate(str(node.params), max_chars=220)),
            ]
            label = _dot_html_label(rows, title=str(node.name), color="#FAD7A0")
        else:
            rows = [("id", str(getattr(node, "id", "?"))), ("type", type(node).__name__)]
            label = _dot_html_label(rows, title=str(getattr(node, "name", "node")), color="#E5E7E9")

        lines.append(f"  {nid} [label={label}];")

    # Edges (compressed by source node_id)
    for node in ir_graph.nodes:
        tgt = f"n{int(node.id)}"
        flat = list(node.input_sources.flatten())

        by_src: dict[int, dict[str, list[int]]] = {}
        # by_src[src_node_id] = {"tgt_ax": [...], "src_idx": [...]}
        for ax_idx, src in enumerate(flat):
            if not isinstance(src, IRSource):
                continue
            if src.is_off():
                continue
            key = int(src.node_id)
            entry = by_src.setdefault(key, {"tgt_ax": [], "src_idx": []})
            entry["tgt_ax"].append(int(ax_idx))
            entry["src_idx"].append(int(src.index))

        # Cap edges per node to avoid unreadable graphs
        src_ids = list(by_src.keys())
        if len(src_ids) > max_edges_per_node:
            src_ids = src_ids[:max_edges_per_node]

        for src_node_id in src_ids:
            entry = by_src[src_node_id]
            if src_node_id == -2:
                src_name = "input"
            elif src_node_id == -3:
                src_name = "const1"
            else:
                src_name = f"n{src_node_id}"

            label = (
                f"axons:{_compress_ranges(entry['tgt_ax'])}\\n"
                f"idx:{_compress_ranges(entry['src_idx'])}\\n"
                f"n={len(entry['tgt_ax'])}"
            )
            lines.append(f"  {src_name} -> {tgt} [label=\"{_truncate(label, max_chars=220)}\"];")

    # Outputs
    out_flat = list(ir_graph.output_sources.flatten())
    by_src_out: dict[int, dict[str, list[int]]] = {}
    for out_i, src in enumerate(out_flat):
        if not isinstance(src, IRSource):
            continue
        if src.is_off():
            continue
        key = int(src.node_id)
        entry = by_src_out.setdefault(key, {"out_i": [], "src_idx": []})
        entry["out_i"].append(int(out_i))
        entry["src_idx"].append(int(src.index))

    for src_node_id, entry in by_src_out.items():
        if src_node_id == -2:
            src_name = "input"
        elif src_node_id == -3:
            src_name = "const1"
        else:
            src_name = f"n{src_node_id}"
        label = f"out:{_compress_ranges(entry['out_i'])}\\nidx:{_compress_ranges(entry['src_idx'])}"
        lines.append(f"  {src_name} -> output [label=\"{_truncate(label, max_chars=220)}\"];")

    lines.append("}")

    with open(out_dot, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# SoftCoreMapping visualization (actual mapped cores)


